import matplotlib.pyplot as plt
import numpy as np

scores = np.load("grid_scores.npy", allow_pickle=True)

#plt.plot(np.a)
scores[:,1][scores[:,1] > 250] = 250 # overfitting
scores[:,2][scores[:,2] > 250] = 250 # overfitting?
#scores[:,2][scores[:,2] > 500] = 500
# print(scores[:,1],scores[:,2])

periods = 30,60,90,120,150,180,210,240
cooldown_types = ["buy", "sell", "both"]
cooldowns = [0,1,2,4,8,16,32]
std_multipliers = [0.5,1,1.5,2,2.5,3,3.5,4]
early_cashouts = ["False", "True"]

divisions = early_cashouts
division_categories = {d:[] for d in divisions}

for label, train, test in scores:
    hps = label.split(",")
    #hp = hps[4].split(" ")[-1]
    hp = str(hps[4].split(" ")[-1])
    division_categories[hp].append((train,test))

for k,v in division_categories.items():
    division_categories[k] = np.array(v).reshape(-1,2)

for k,v in division_categories.items():
    print(v[:,0], v[:,1])
    plt.scatter(v[:,0], v[:,1], label=str(k))
    #plt.plot()

plt.plot(np.arange(90,150), np.arange(90, 150))
plt.legend()

plt.show()




    #print(label, train, test)


# plt.scatter(scores[:,1], scores[:,2])
# plt.plot(np.arange(90,150), np.arange(90, 150))
#
# for label, train, test in scores:
#     plt.scatter(train, test, label=label)

# plt.xlabel("Train")
# plt.ylabel("Test")
# for label, train, test in scores:
#     if abs(train-120) < 5 and abs(test-120) < 5:
#         print(label, train, test)
# plt.show()