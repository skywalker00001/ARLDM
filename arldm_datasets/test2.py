predictions = [
    [[1], [2, 3]],
    [[4]],
    [[5, 6], [7, 8, 9]]
]

images = [elem for sublist in predictions for elem in sublist[-1]]
print(images)