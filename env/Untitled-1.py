
def counting(data):
    # Creates 2D list of size max number in the array
    counts = [0 for i in range(max(data)+1)]
    print(counts)
    # Finds the "counts" for each individual number
    for value in data:
        counts[value] += 1
    print(counts)    
    # Finds the cumulative sum counts
    for index in range(1, len(counts)):
        counts[index] = counts[index-1] + counts[index]
    print(counts)
    # Sorting Phase
    arr = [0 for loop in range(len(data))]
    for value in data:
        index = counts[value] - 1
        arr[index] = value
        counts[value] -= 1
    return arr
    
data = [6,2,1,3,1,3,2]
print(counting(data))