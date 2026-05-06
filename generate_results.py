import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'repo', 'Chapter 3'))

from hw3_1_naive_dqn import train_naive_dqn, train_experience_replay
from hw3_2_advanced_dqn import train_double_dqn, train_dueling_dqn

os.makedirs('static/images', exist_ok=True)
plt.style.use('dark_background')

print("Running Naive DQN...")
_, loss1 = train_naive_dqn(epochs=500)
plt.figure()
plt.plot(loss1, color='#ff4757')
plt.title("Naive DQN Loss (Static Mode)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("static/images/naive_dqn_loss.png", transparent=True)
plt.close()

print("Running Experience Replay DQN...")
_, loss2 = train_experience_replay(epochs=500)
plt.figure()
plt.plot(loss2, color='#2ed573')
plt.title("Experience Replay Loss (Static Mode)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("static/images/er_dqn_loss.png", transparent=True)
plt.close()

print("Running Double DQN...")
_, loss3 = train_double_dqn(epochs=500)
plt.figure()
plt.plot(loss3, color='#1e90ff')
plt.title("Double DQN Loss (Player Mode)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("static/images/double_dqn_loss.png", transparent=True)
plt.close()

print("Running Dueling DQN...")
_, loss4 = train_dueling_dqn(epochs=500)
plt.figure()
plt.plot(loss4, color='#ffa502')
plt.title("Dueling DQN Loss (Player Mode)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("static/images/dueling_dqn_loss.png", transparent=True)
plt.close()

print("Done generating plots.")
