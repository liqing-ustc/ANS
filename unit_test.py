import sys
sys.setrecursionlimit(int(1e4))
sys.path.insert(0, "./semantics/dreamcoder")

from dreamcoder.domains.hint.hintPrimitives import McCarthyPrimitives
_ = McCarthyPrimitives()
from dreamcoder.program import Program
from semantics.semantics import ProgramWrapper
pg = Program.parse("(lambda (lambda (fix2 $1 $0 (lambda (lambda (lambda (if0 $0 $1 ($2 (incr $1) (decr0 $0)))))))))")
pg = ProgramWrapper(pg)
print(pg(1, 2))
# pg = Program.parse("(lambda (lambda (fix2 $1 $0 (lambda (lambda (lambda (if0 $0 $1 ($2 $0 (decr0 $1)))))))))")
pg = Program.parse("(lambda (lambda (fix2 $1 $0 (lambda (lambda (lambda (if0 $0 $1 (incr ($2 $1 (decr0 $0))))))))))")
pg = ProgramWrapper(pg)
print(pg(1, 1))
pass