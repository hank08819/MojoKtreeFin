# Mojo Extra Trees Regressor — CALLS-only and PUTS-only IV prediction
# d=7: [strike, spot, DTE, bid, ask, volume, lastPrice]
# Mojo: 200K train; sklearn comparison uses 20K (extra_trees_sklearn_iv7.py)

from std.collections import List
from std.time import perf_counter_ns
from std.pathlib import Path

comptime DIMS=7; comptime NRUNS=3
comptime LEAF=10; comptime MAX_DEPTH=20; comptime MAX_FEAT=3; comptime N_TREES=20

struct Rng(Copyable):
    var state:UInt64
    def __init__(out self, seed:UInt64): self.state=seed^6364136223846793005
    def copy(self) -> Rng: return Rng(self.state)
    def next(mut self) -> UInt64:
        self.state=self.state*6364136223846793005+1442695040888963407
        return self.state
    def next_float(mut self) -> Float32:
        return Float32(Int(self.next()&0xFFFF))/Float32(65536.0)
    def next_int(mut self, n:Int) -> Int:
        return Int(self.next()%UInt64(n))

struct TNode(Copyable):
    var feat:Int; var thr:Float32; var left:Int; var right:Int; var leaf_mean:Float32
    def __init__(out self):
        self.feat=-1; self.thr=Float32(0); self.left=-1; self.right=-1; self.leaf_mean=Float32(0)
    def copy(self) -> TNode:
        var n=TNode(); n.feat=self.feat; n.thr=self.thr
        n.left=self.left; n.right=self.right; n.leaf_mean=self.leaf_mean; return n^

struct ExTreReg:
    var Xtr:List[Float32]; var Ytr:List[Float32]
    var Xte:List[Float32]; var Yte:List[Float32]
    var n_tr:Int; var n_te:Int
    var all_nodes:List[TNode]; var tree_roots:List[Int]; var idx_buf:List[Int]; var rng:Rng

    def __init__(out self):
        self.n_tr=0; self.n_te=0
        self.Xtr=List[Float32](); self.Ytr=List[Float32]()
        self.Xte=List[Float32](); self.Yte=List[Float32]()
        self.all_nodes=List[TNode](); self.tree_roots=List[Int]()
        self.idx_buf=List[Int](); self.rng=Rng(UInt64(42))

    def reset(mut self):
        self.n_tr=0; self.n_te=0
        self.Xtr=List[Float32](); self.Ytr=List[Float32]()
        self.Xte=List[Float32](); self.Yte=List[Float32]()
        self.all_nodes=List[TNode](); self.tree_roots=List[Int]()
        self.idx_buf=List[Int](); self.rng=Rng(UInt64(42))

    def load_f32_X(mut self, path:String, is_train:Bool) raises:
        var raw=Path(path).read_bytes()
        var nr=Int(raw[0])|(Int(raw[1])<<8)|(Int(raw[2])<<16)|(Int(raw[3])<<24)
        var nc=Int(raw[4])|(Int(raw[5])<<8)|(Int(raw[6])<<16)|(Int(raw[7])<<24)
        var ptr=(raw.unsafe_ptr()+8).bitcast[Float32]()
        for i in range(nr*nc):
            if is_train: self.Xtr.append(ptr[i])
            else:        self.Xte.append(ptr[i])
        if is_train: self.n_tr=nr
        else:        self.n_te=nr

    def load_f32_y(mut self, path:String, is_train:Bool) raises:
        var raw=Path(path).read_bytes()
        var n=Int(raw[0])|(Int(raw[1])<<8)|(Int(raw[2])<<16)|(Int(raw[3])<<24)
        var ptr=(raw.unsafe_ptr()+4).bitcast[Float32]()
        for i in range(n):
            if is_train: self.Ytr.append(ptr[i])
            else:        self.Yte.append(ptr[i])

    def _build(mut self, lo:Int, hi:Int, depth:Int) -> Int:
        var cnt=hi-lo; var nd=TNode()
        if cnt<=LEAF or depth>=MAX_DEPTH:
            var s=Float32(0)
            for i in range(lo,hi): s+=self.Ytr[self.idx_buf[i]]
            nd.leaf_mean=s/Float32(cnt)
            self.all_nodes.append(nd.copy()); return len(self.all_nodes)-1
        var bfeat=0; var bthr=Float32(0); var found=False
        var ptr_x=self.Xtr.unsafe_ptr()
        for _ in range(MAX_FEAT*3):
            var feat=self.rng.next_int(DIMS)
            var fmin=Float32(3.4e38); var fmax=Float32(-3.4e38)
            for i in range(lo,hi):
                var v=ptr_x[self.idx_buf[i]*DIMS+feat]
                if v<fmin: fmin=v
                if v>fmax: fmax=v
            if fmax<=fmin: continue
            bthr=fmin+(fmax-fmin)*self.rng.next_float()
            bfeat=feat; found=True; break
        if not found:
            var s=Float32(0)
            for i in range(lo,hi): s+=self.Ytr[self.idx_buf[i]]
            nd.leaf_mean=s/Float32(cnt)
            self.all_nodes.append(nd.copy()); return len(self.all_nodes)-1
        var s=lo; var ptr_x2=self.Xtr.unsafe_ptr()
        for i in range(lo,hi):
            if ptr_x2[self.idx_buf[i]*DIMS+bfeat]<=bthr:
                var t=self.idx_buf[s]; self.idx_buf[s]=self.idx_buf[i]; self.idx_buf[i]=t; s+=1
        if s==lo or s==hi:
            var sm=Float32(0)
            for i in range(lo,hi): sm+=self.Ytr[self.idx_buf[i]]
            nd.leaf_mean=sm/Float32(cnt)
            self.all_nodes.append(nd.copy()); return len(self.all_nodes)-1
        nd.feat=bfeat; nd.thr=bthr
        self.all_nodes.append(nd.copy()); var me=len(self.all_nodes)-1
        var li=self._build(lo, s, depth+1)
        var ri=self._build(s, hi, depth+1)
        var tmp=self.all_nodes[me].copy(); tmp.left=li; tmp.right=ri
        self.all_nodes[me]=tmp.copy(); return me

    def build_forest(mut self):
        for t in range(N_TREES):
            self.idx_buf=List[Int]()
            for i in range(self.n_tr): self.idx_buf.append(i)
            self.tree_roots.append(len(self.all_nodes))
            var _r=self._build(0, self.n_tr, 0)

    @always_inline
    def _traverse(self, xi:Int, root:Int) -> Float32:
        var ni=root; var ptr_x=self.Xte.unsafe_ptr()
        while True:
            var nd=self.all_nodes[ni].copy()
            if nd.left==-1: return nd.leaf_mean
            if ptr_x[xi*DIMS+nd.feat]<=nd.thr: ni=nd.left
            else: ni=nd.right

    @always_inline
    def predict(self, xi:Int) -> Float32:
        var s=Float32(0)
        for t in range(N_TREES): s+=self._traverse(xi, self.tree_roots[t])
        return s/Float32(N_TREES)

    def run_benchmark(mut self, label:String) raises:
        print("\n===",label,"  n_train=",self.n_tr,"  n_test=",self.n_te," ===")
        var tb=perf_counter_ns()
        self.build_forest()
        var build_ms=Float64(Int(perf_counter_ns()-tb))/1e6
        print("  Nodes total =",len(self.all_nodes))
        print("  Build ms    =",build_ms)
        var infer_ns=Int(0); var last_sse=Float64(0); var last_sae=Float64(0)
        for run in range(NRUNS):
            var t0=perf_counter_ns(); var sse=Float64(0); var sae=Float64(0)
            for i in range(self.n_te):
                var pred=self.predict(i)
                var err=Float64(pred)-Float64(self.Yte[i])
                sse+=err*err
                if err>=0.0: sae+=err
                else: sae+=(-err)
            infer_ns+=Int(perf_counter_ns()-t0)
            if run==NRUNS-1: last_sse=sse; last_sae=sae
        var infer_s=Float64(infer_ns)/Float64(NRUNS)/1e9
        print("  Inference s =",infer_s)
        print("  RMSE        =",(last_sse/Float64(self.n_te))**0.5)
        print("  MAE         =",last_sae/Float64(self.n_te))

def main() raises:
    var base=String("/Users/henry_han/LHP2/MOJO/extended_benchmark/data/")
    var et=ExTreReg()

    # ── CALLS only ────────────────────────────────────────────────────────────
    et.load_f32_X(base+"CALLS/X_train200k.bin", True)
    et.load_f32_X(base+"CALLS/X_test50k.bin",   False)
    et.load_f32_y(base+"CALLS/y_train200k.bin",  True)
    et.load_f32_y(base+"CALLS/y_test50k.bin",    False)
    et.run_benchmark(String("Mojo ExTre CALLS IV  DIMS=7  N_TREES=20"))

    # ── PUTS only ─────────────────────────────────────────────────────────────
    et.reset()
    et.load_f32_X(base+"PUTS/X_train200k.bin", True)
    et.load_f32_X(base+"PUTS/X_test50k.bin",   False)
    et.load_f32_y(base+"PUTS/y_train200k.bin",  True)
    et.load_f32_y(base+"PUTS/y_test50k.bin",    False)
    et.run_benchmark(String("Mojo ExTre PUTS  IV  DIMS=7  N_TREES=20"))
