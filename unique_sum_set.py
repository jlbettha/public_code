import numpy as np
import itertools

# how to construct a sequence of positive integers with a unique sum of any previous numbers. Known as the golomb ruler problem (https://en.wikipedia.org/wiki/Golomb_ruler) NP hard
N = 100
final_list = []  # <- empty set
seen_list = []  # <- emtpy set  # Sums seen so far
open_list = []  # <- empty set  # Sums ending at the last element

numbers = [1, 2, 5]
result = [
    np.sum(seq)
    for i in range(len(numbers), 0, -1)
    for seq in itertools.combinations(numbers, i)
]

print(numbers)
print(result)
exit()

for x in np.arange(1, N + 1, 1):
    if x in seen_list:
        # quick fail
        continue  # with next x

    # Build new set
    pending_list = []  # <- empty set
    pending_list.append(x)  # add x to Pending
    for s in open_list:
        pending_list.append(s + x)  # add (s+x) to Pending

    # Check if these numbers are all unique
    if not list(
        set(pending_list) & set(seen_list)
    ):  # if (pending_list intersection seen_list) is []:
        # If so, we have a new result
        final_list.append(x)  # yield x
        print(x)
        open_list.extend(pending_list)
        seen_list.extend(
            list(set(pending_list) | set(seen_list))
        )  # seen_list union pending_list)

print(final_list)
# From <https://stackoverflow.com/questions/6936438/how-to-construct-a-sequence-of-positive-integers-with-a-unique-sum-of-any-contig>

# The first 30 values (when selecting always the smallest possible value for the next difference) are
# test30 = [1, 2, 4, 5, 8, 10, 14, 21, 15, 16, 26, 25, 34, 22, 48, 38, 71, 40, 74, 90, 28, 69, 113, 47, 94, 54, 46, 143, 153, 83]
## This list is not the one i want. i.e, 5 is the sum of 1+4 and 8 is the sum of 1+2+5.
