// Bell pair: measure both qubits -> 00 or 11, never 01 or 10.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
