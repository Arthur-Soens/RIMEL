import sys
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


def main() :
    print(sys.argv)
    dir = sys.argv[1]
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(len(files))
    notes = []
    for file_name in files :
        file = open(dir+"/"+file_name, "r")
        lines = file.readlines()
        file.close()

        for line in lines:
            if "Your code has been rated at" in line:
                note = round(float(line.split(" ")[6].split("/")[0]),1)
                notes.append(note)

    notes.sort()
    print(notes)
    print(notes[-1])
    rates = {}
    rates_range = {
        '0-3' : 0,
        '3-5' : 0,
        '5-8': 0,
        '8-10' : 0
    }
    for note in notes :
        if (note in rates) :
            rates[note] += 1
        else :
            rates[note] = 1
        if note <=3 :
            rates_range['0-3'] += 1
        elif note <= 5:
            rates_range['3-5'] += 1
        elif note <=8 :
            rates_range['5-8'] += 1
        else :
            rates_range['8-10'] += 1

    print(rates.keys())
    plt.title('Mean rate of notebook')
    plt.plot(rates.keys(),rates.values(), 'o')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.xlabel('Rate')
    plt.ylabel('Number of notebook')
    plt.legend()
    plt.show()

    plt.title('Number of notebook per rank ranges')
    plt.bar(rates_range.keys(), rates_range.values(),width=0.4)
    plt.xlabel('Rate ranges')
    plt.ylabel('Number of notebook')
    plt.legend()
    plt.show()

main()