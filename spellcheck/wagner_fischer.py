import os
import time


def load_dictionary(file_path):
    with open(file_path) as file:
        return [line.strip() for line in file]


def wagner_fischer(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 > len_s2:
        s1, s2 = s2, s1
        len_s1, len_s2 = len_s2, len_s1

    current_row = range(len_s1 + 1)
    for i in range(1, len_s2 + 1):
        previous_row, current_row = current_row, [i] + [0] * len_s1
        for j in range(1, len_s1 + 1):
            add, delete, change = (
                previous_row[j] + 1,
                current_row[j - 1] + 1,
                previous_row[j - 1],
            )
            if s1[j - 1] != s2[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[len_s1]


def spell_check(word, dictionary):
    suggestions = []

    for correct_word in dictionary:
        distance = wagner_fischer(word, correct_word)
        suggestions.append((correct_word, distance))

    suggestions.sort(key=lambda x: x[1])
    return suggestions


def main() -> None:
    # Example Usage
    topx: int = 7
    pwd = os.path.dirname(os.path.abspath(__file__))
    dictionary = load_dictionary(os.path.join(pwd, "words.txt"))
    misspelled_word = "wrlod"
    suggestions = spell_check(misspelled_word, dictionary)
    print(f"Top {topx} suggestions for '{misspelled_word}':")
    for word, distance in suggestions[:topx]:
        print(f"{word} (Distance: {distance})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time() - t0:.3f} seconds.")
