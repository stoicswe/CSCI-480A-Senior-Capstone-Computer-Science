import math, numpy, random, time # normal python stuff

from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, S, T, X, CNOT, get_inverse, Measure  # import the operations we want to perform

import projectq.setups.ibm
from projectq.backends import IBMBackend

eng = MainEngine() #uncomment to simulate on your own computer
#eng = MainEngine(IBMBackend(use_hardware=True, num_runs=1, verbose=True, user=None, password=None)) # uncomment to run on the quantum experience

print("\n\n\n\n===== Welcome to Cat/Box/Scissors! =====\n\n")
print("  ~~ A game by the Decodoku project ~~ \n\n")
print("When in doubt, press any key to continue!")
input()
print("You and your opponent choose one of two possible moves.")
input()
print("You win if your moves are the same.")
input()
print("Your opponent wins if they are different.")
input()

# get human player to choose opponent
chosen = 0
while (chosen==0):
	opponent = int( input("\nWhich qubit will be your opponent? (1,2,3, or 4)\n") )
	if ( (opponent >= 1) & (opponent <= 4) ):
		chosen = 1
	else:
		print("u wot m8? Try that again.")

# here 1 and 2 mean qubits 0 and 1, so change accordingly
if (opponent<3):
	opponent = opponent - 1

# referee is always qubit 2

# get human player to choose move
chosen = 0
while (chosen==0):
	humanMove = input("\nChoose your move (s or sdg)\n")
	if ( (humanMove == "s") | (humanMove == "sdg") ):
		chosen = 1
	else:
		print("u wot m8? Try that again.")

print("\nWe'll now send your move to the quantum referee at IBM.")
input()
print("It will take your opponents move and compare them.")
input()
print("But first you'll have to sign in...\n")

# prepare qubits
qubits = eng.allocate_qureg(5)

# referee decides things in the X basis, so we prepare it in the |+> state
H | qubits[2]

# implement human move
if (humanMove == "s"):
	S | qubits[2]
else:
	get_inverse(S) | qubits[2]

# opponent qubit is prepared in state |+> to randomly decide the move to make
H | qubits[opponent]

# to implement the quantum move, first do an S in all cases
S | qubits[2]
# then use a controlled-Z to make it into a Sdg if that's what the quantum player chooses
H | qubits[2]
CNOT | (qubits[opponent], qubits[2])
H | qubits[2]

# quantum player wins if the moves where different, which would leave referee in the |+> state
# human player wins if the moves were the same
# we measure in the X basis to see
H | qubits[2]
Measure | qubits[2]

eng.flush()  # flush all gates (and execute measurements)

print("\nThe referee has decided...\n")
time.sleep(1)
if int(qubits[2]):
	print("You win!\n")
else:
	print("You Lose!\n")