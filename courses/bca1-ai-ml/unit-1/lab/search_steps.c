/*
 * Unit 1 lab · Program 2: linear search vs binary search
 *
 * Problem : roll numbers 1..n are stored in sorted order. Find a given roll number.
 * Input   : n (1 <= n <= 1000000) and the roll number to find (key)
 * Output  : position of key (or "not found") and the comparisons each method used
 * Pre     : the list is sorted in increasing order (binary search needs this!)
 * Post    : if key is in the list, list[position] == key
 *
 * Compile : gcc search_steps.c -o search_steps
 */
#include <stdio.h>

#define MAX 1000000
int list[MAX];

/* read one whole number in [lo, hi]; ask again until it is valid */
long read_in_range(const char *prompt, long lo, long hi)
{
    long x;
    printf("%s (%ld to %ld): ", prompt, lo, hi);
    while (scanf("%ld", &x) != 1 || x < lo || x > hi) {
        int c;
        while ((c = getchar()) != '\n' && c != EOF)
            ;
        if (c == EOF) return lo;
        printf("Invalid, try again: ");
    }
    return x;
}

int main(void)
{
    long n = read_in_range("How many students n", 1, MAX);
    for (long i = 0; i < n; i++)
        list[i] = (int)(i + 1);               /* roll numbers 1, 2, 3, ... n (sorted) */
    long key = read_in_range("Roll number to find", 1, 2 * (long)MAX);

    /* ---- linear search: check every element from the start ---- */
    long linPos = -1, linSteps = 0;
    for (long i = 0; i < n; i++) {
        linSteps++;
        if (list[i] == key) { linPos = i; break; }
    }

    /* ---- binary search: look at the middle, throw away half each time ---- */
    long lo = 0, hi = n - 1, binPos = -1, binSteps = 0;
    while (lo <= hi) {
        long mid = (lo + hi) / 2;
        binSteps++;
        if (list[mid] == key) { binPos = mid; break; }
        else if (list[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }

    if (linPos >= 0) printf("\nFound at position %ld\n", linPos + 1);
    else             printf("\nNot found\n");
    printf("Linear search : %ld comparisons\n", linSteps);
    printf("Binary search : %ld comparisons\n", binSteps);
    printf("Both methods agree: %s\n", linPos == binPos ? "yes" : "NO, check the code");
    return 0;
}
