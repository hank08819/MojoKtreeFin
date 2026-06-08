# Mojo KD-KNN d=24 — all 7 non-BAC financial tickers on ARM64
from std.collections import List
from std.time import perf_counter_ns
from std.pathlib import Path

comptime DIMS=24; comptime SW=8; comptime KN=5; comptime NRUNS=10; comptime LEAF=10

struct Node(Copyable):
    var axis:Int; var split:Float32; var left:Int; var right:Int; var lo:Int; var hi:Int
    def __init__(out self):
        self.axis=-1; self.split=Float32(0); self.left=-1; self.right=-1; self.lo=0; self.hi=0

struct KNN:
    var Xtr:List[Float32]; var Ytr:List[Int32]
    var Xte:List[Float32]; var Yte:List[Int32]
    var n_tr:Int; var n_te:Int
    var idx:List[Int]; var nodes:List[Node]
    var hdists:List[Float32]; var hlbls:List[Int32]

    def __init__(out self):
        self.n_tr=0; self.n_te=0
        self.Xtr=List[Float32](); self.Ytr=List[Int32]()
        self.Xte=List[Float32](); self.Yte=List[Int32]()
        self.idx=List[Int](); self.nodes=List[Node]()
        self.hdists=List[Float32](); self.hlbls=List[Int32]()
        for _ in range(KN): self.hdists.append(Float32(1e38)); self.hlbls.append(Int32(0))

    def load_f32(mut self, path:String, is_train:Bool) raises:
        var raw=Path(path).read_bytes()
        var nr=Int(raw[0])|(Int(raw[1])<<8)|(Int(raw[2])<<16)|(Int(raw[3])<<24)
        var nc=Int(raw[4])|(Int(raw[5])<<8)|(Int(raw[6])<<16)|(Int(raw[7])<<24)
        var ptr=(raw.unsafe_ptr()+8).bitcast[Float32]()
        for i in range(nr*nc):
            if is_train: self.Xtr.append(ptr[i])
            else:        self.Xte.append(ptr[i])
        if is_train: self.n_tr=nr
        else:        self.n_te=nr

    def load_i32(mut self, path:String, is_train:Bool) raises:
        var raw=Path(path).read_bytes()
        var n=Int(raw[0])|(Int(raw[1])<<8)|(Int(raw[2])<<16)|(Int(raw[3])<<24)
        var ptr=(raw.unsafe_ptr()+4).bitcast[Int32]()
        for i in range(n):
            if is_train: self.Ytr.append(ptr[i])
            else:        self.Yte.append(ptr[i])

    @always_inline
    def qdist(self, ti:Int, qi:Int) -> Float32:
        var a=self.Xtr.unsafe_ptr()+ti*DIMS; var b=self.Xte.unsafe_ptr()+qi*DIMS
        var d0=b.load[width=SW](0)-a.load[width=SW](0)
        var d1=b.load[width=SW](8)-a.load[width=SW](8)
        var d2=b.load[width=SW](16)-a.load[width=SW](16)
        return (d0*d0).reduce_add()+(d1*d1).reduce_add()+(d2*d2).reduce_add()

    def heap_reset(mut self):
        for i in range(KN): self.hdists[i]=Float32(1e38); self.hlbls[i]=Int32(0)

    def heap_push(mut self, d:Float32, lbl:Int32):
        if d>=self.hdists[0]: return
        self.hdists[0]=d; self.hlbls[0]=lbl
        var i=0
        while True:
            var l=2*i+1; var r=2*i+2; var mx=i
            if l<KN and self.hdists[l]>self.hdists[mx]: mx=l
            if r<KN and self.hdists[r]>self.hdists[mx]: mx=r
            if mx==i: break
            var td=self.hdists[i]; self.hdists[i]=self.hdists[mx]; self.hdists[mx]=td
            var tl=self.hlbls[i]; self.hlbls[i]=self.hlbls[mx]; self.hlbls[mx]=tl
            i=mx

    def heap_vote(self) -> Int32:
        var c=0
        for i in range(KN): c+=Int(self.hlbls[i])
        return Int32(1) if c*2>=KN else Int32(0)

    def build_tree(mut self):
        for i in range(self.n_tr): self.idx.append(i)
        _ = self._build(0, self.n_tr)

    def _build(mut self, lo:Int, hi:Int) -> Int:
        var nd=Node(); nd.lo=lo; nd.hi=hi
        if hi-lo<=LEAF: self.nodes.append(nd.copy()); return len(self.nodes)-1
        var cnt=hi-lo; var bax=0; var bv=Float32(-1.0)
        for ax in range(DIMS):
            var m=Float32(0.0)
            for i in range(lo,hi): m+=self.Xtr[self.idx[i]*DIMS+ax]
            m/=Float32(cnt); var v=Float32(0.0)
            for i in range(lo,hi):
                var d2=self.Xtr[self.idx[i]*DIMS+ax]-m; v+=d2*d2
            if v>bv: bv=v; bax=ax
        var mid=lo+cnt//2; var l=lo; var h=hi-1
        while l<h:
            var pi=l+(h-l)//2; var pv=self.Xtr[self.idx[pi]*DIMS+bax]
            var t=self.idx[pi]; self.idx[pi]=self.idx[h]; self.idx[h]=t
            var s=l
            for j in range(l,h):
                if self.Xtr[self.idx[j]*DIMS+bax]<pv:
                    t=self.idx[j]; self.idx[j]=self.idx[s]; self.idx[s]=t; s+=1
            t=self.idx[s]; self.idx[s]=self.idx[h]; self.idx[h]=t
            if s==mid: break
            elif s<mid: l=s+1
            else: h=s-1
        nd.axis=bax; nd.split=self.Xtr[self.idx[mid]*DIMS+bax]
        self.nodes.append(nd.copy()); var me=len(self.nodes)-1
        var li=self._build(lo,mid); var ri=self._build(mid,hi)
        var tmp=self.nodes[me].copy(); tmp.left=li; tmp.right=ri
        self.nodes[me]=tmp.copy(); return me

    def kd_predict(mut self, qi:Int) -> Int32:
        self.heap_reset(); self._search(0,qi); return self.heap_vote()

    def _search(mut self, ni:Int, qi:Int):
        var nd=self.nodes[ni].copy()
        if nd.axis==-1:
            for i in range(nd.lo,nd.hi):
                self.heap_push(self.qdist(self.idx[i],qi),self.Ytr[self.idx[i]])
            return
        var qv=self.Xte[qi*DIMS+nd.axis]; var diff=qv-nd.split
        if diff<=0.0:
            self._search(nd.left,qi)
            if diff*diff<self.hdists[0]: self._search(nd.right,qi)
        else:
            self._search(nd.right,qi)
            if diff*diff<self.hdists[0]: self._search(nd.left,qi)

def run_ticker(base:String, label:String) raises:
    print("\n=== "+label+" ===")
    var knn=KNN()
    knn.load_f32(base+"train_X.bin",True); knn.load_f32(base+"test_X.bin",False)
    knn.load_i32(base+"train_y.bin",True); knn.load_i32(base+"test_y.bin",False)
    print("  n_train:",knn.n_tr,"  n_test:",knn.n_te)
    var tb=perf_counter_ns(); knn.build_tree()
    print("  Build:",Float64(Int(perf_counter_ns()-tb))/1e6,"ms  nodes:",len(knn.nodes))
    var kd_ns=Int(0); var kd_ok=Int(0)
    for run in range(NRUNS):
        var t0=perf_counter_ns(); var ok=Int(0)
        for i in range(knn.n_te):
            if knn.kd_predict(i)==knn.Yte[i]: ok+=1
        kd_ns+=Int(perf_counter_ns()-t0)
        if run==NRUNS-1: kd_ok=ok
    var kd_s=Float64(kd_ns)/Float64(NRUNS)/1e9
    var acc=Float64(kd_ok)/Float64(knn.n_te)
    print("  Mojo KD mean:",kd_s,"s   acc:",acc)

def main() raises:
    var root=String("/Users/henry_han/LHP2/MOJO/extended_benchmark/data/")
    run_ticker(root+"CPRI/",   "CPRI")
    run_ticker(root+"SPY/",    "SPY")
    run_ticker(root+"QQQ/",    "QQQ")
    run_ticker(root+"EURUSD/", "EURUSD")
    print("\n=== Done (CPRI+ARM64) ===")
