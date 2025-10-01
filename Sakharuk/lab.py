import random;
import matplotlib.pyplot as plt

def maxLengthOfSerie(totalExp = 100000, p = 0.5):
    maxLength = 0
    for i in range(totalExp):
        res = exp100(p=p)
        currentLength = 0
        for item in res:
            if(item == 0):
                currentLength += 1
            else:
                if(currentLength > maxLength):
                    maxLength = currentLength
                currentLength = 0
        if(currentLength > maxLength):
            maxLength = currentLength
    return maxLength

def exp100 (n=100, p=0.5):
    res = []
    for i in range(n):
        if(random.random() < p):
            res.append(0)
        else:
            res.append(1)
    return res

def multipleExp100(n=100, totalExp=100000, p=0.5):
    eagles=[]
    successful = 0
    for exp in range(totalExp):
        res100 = exp100(n,p)
        eaglesNum = 0

        isSeries = False
        currentSerie = 0

        for i in range(n):
            if(res100[i] == 0):
                currentSerie += 1
                eaglesNum += 1
            else:
                currentSerie = 0
            if(not isSeries and currentSerie >= 5):
                isSeries = True
        
        eagles.append(eaglesNum)
        if(isSeries):
            successful += 1

    return successful, eagles

def numEaglesGreater60(res, totalExp=100000):
    num60 = 0
    for i in range(totalExp):
        if(res[i] > 60):
            num60 += 1
    return num60

def intervals(res, totalExp = 100000):
    probabilities = []
    for i in range(10):
        numInInterval = 0
        for j in range(totalExp):
            if(i+1 == 10):
                if(res[j] <= (i+1)*10 and res[j] >= (i)*10):
                    numInInterval += 1
            elif(res[j] < (i+1)*10 and res[j] >= (i)*10):
                numInInterval += 1
        probabilities.append(numInInterval / totalExp)
    return probabilities
    
def probability95prc(res, totalExp=100000, p = 0.5):
    newP = p
    right = p * 100
    left = p * 100 + 1
    while(newP < 0.95):
        right += 1
        left -= 1
        numInInterval = 0
        for i in range(totalExp):
            if(res[i] >= left and res[i] < right):
                numInInterval += 1
        newP = numInInterval / totalExp
    return left, right

def setGraphics():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    for item in x:
        successful, res = multipleExp100(p=item)
        y1.append(sum(res)/len(res))

        left,right = probability95prc(res, p=item)
        y2.append(right-left)

        y3.append(successful/len(res))

        maxLength = maxLengthOfSerie(p=item)
        y4.append(maxLength)

    fig, ((gr1, gr2), (gr3, gr4)) = plt.subplots(2, 2, figsize=(15, 10))
    # first
    gr1.plot(x,y1,'mo-',linewidth=2, markersize=6)
    gr1.set_title("Average num of eagles", fontsize=14)
    gr1.set_xlabel("Probability of eagles", fontsize=12)
    gr1.set_ylabel("Average of eagles in 100 tries", fontsize=12)
    gr1.grid(True, alpha=0.4,linestyle='-')
    
    # second;
    gr2.plot(x,y2,'mo-',linewidth=2, markersize=6)
    gr2.set_title("Probability 95 in interval", fontsize=14)
    gr2.set_xlabel("Probability of eagles", fontsize=12)
    gr2.set_ylabel("Length of interval", fontsize=12)
    gr2.grid(True, alpha=0.4,linestyle='-')

    #third
    gr3.plot(x,y3,'mo-',linewidth=2, markersize=6)
    gr3.set_title("Probability of serie", fontsize=14)
    gr3.set_xlabel("Probability of eagles", fontsize=12)
    gr3.set_ylabel("Probability of serie", fontsize=12)
    gr3.grid(True, alpha=0.4,linestyle='-')

    #forth
    gr4.plot(x,y4,'mo-',linewidth=2, markersize=6)
    gr4.set_title("Max series", fontsize=14)
    gr4.set_xlabel("Probability of eagles", fontsize=12)
    gr4.set_ylabel("Max series", fontsize=12)
    gr4.grid(True, alpha=0.4,linestyle='-')

    plt.tight_layout()
    plt.show()

def main():
    n = 100
    totalExp = 100000
    successful, res = multipleExp100(n, totalExp)
    print(f"\n1) Average number of eagles in the number of experiments: {sum(res) / totalExp}\n")

    num60 = numEaglesGreater60(res, totalExp)
    print(f"2) With this probability, you can get a number of heads greater than 60 with a given number of experiments {(num60 / totalExp):.6f}\n")

    probabilitiesInIntervals = intervals(res, totalExp)
    print(f"3) With this probability, the number of heads corresponding to these intervals will fall out for the given number of experiments")
    for i in range(10):
        if(i != 10):
            print(f"[{(i)*10},{(i+1)*10}): {probabilitiesInIntervals[i]:.6f}")
        else:
            print(f"[{(i)*10},{(i+1)*10}]: {probabilitiesInIntervals[i]:.6f}")

    left, right = probability95prc(res, totalExp)
    print(f"\n4) The probability of getting heads is 0.95 on the interval [{left}, {right})")

    print(f"\n5) With probability {successful / totalExp} there will be at least one series of 5 heads in the given number of experiments\n")

    print(f"6) The coin is not symmetrical. Graphics for points 1,4,5 according to the given conditions\n")
    setGraphics()

if __name__ == "__main__":
    main()