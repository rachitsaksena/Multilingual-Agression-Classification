def dist(source, target):
    """trying to match word source to target word"""
    dp = [[0 for i in range(len(source)+1)] for i in range(2)]

    for i in range(len(source)+1): dp[0][i] = i

    for i in range(len(target)+1):
        for j in range(len(source)+1):
            if j==0: dp[i%2][j] = i
            elif source[j-1] == target[i-1]: dp[i%2][j] = dp[(i-1)%2][j-1]
            else: dp[i%2][j] = 1+min(dp[(i-1)%2][j], dp[(i)%2][j-1], dp[(i-1)%2][j-1])
        
    return dp[len(target)%2][len(source)]

def main():
    w1 = input("Enter source word: ")
    w2 = input("Enter target word: ")
    print(dist(w1, w2))

if __name__=="__main__":
    main()