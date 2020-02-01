text = open("sample.txt", "r")


d = dict()


for line in text:

    line = line.strip()


    line = line.lower()


    words = line.split(" ")


    for word in words:
        # Check if the word is already in dictionary
        if word in d:
            # Increment count of word by 1
            d[word] = d[word] + 1
        else:
            # Add the word to dictionary with count 1
            d[word] = 1

# Print the contents of dictionary
for key in list(d.keys()):
    print(key, ":", d[key])