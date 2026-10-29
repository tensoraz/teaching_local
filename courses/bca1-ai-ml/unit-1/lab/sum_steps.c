/*
 * Unit 1 lab · Program 1: two algorithms for the same problem
 *
 * Problem : find S = 1 + 2 + 3 + ... + n
 * Input   : one whole number n, with 1 <= n <= 1000000
 * Output  : S from both algorithms, and how many additions each one needed
 * Pre     : n has been validated (1 <= n <= 1000000)
 * Post    : both answers are equal, and equal to n(n+1)/2
 *
 * Compile : gcc sum_steps.c -o sum_steps
 * Run     : ./sum_steps        (Windows: sum_steps)
 */
#include <stdio.h>

int main(void)
{
    long long n;

    /* ---- input + validation: keep asking until the input is acceptable ---- */
    printf("Enter n (1 to 1000000): ");
    while (scanf("%lld", &n) != 1 || n < 1 || n > 1000000) {
        int c;
        while ((c = getchar()) != '\n' && c != EOF)   /* throw away the bad input */
            ;
        if (c == EOF) return 1;
        printf("Invalid. n must be a whole number from 1 to 1000000. Try again: ");
    }

    /* ---- Algorithm A: add the numbers one by one ---- */
    long long sumA = 0, stepsA = 0;
    for (long long i = 1; i <= n; i++) {
        sumA = sumA + i;
        stepsA++;
    }

    /* ---- Algorithm B: Gauss's formula ---- */
    long long sumB = n * (n + 1) / 2;
    long long stepsB = 1;            /* one multiplication-and-division, whatever n is */

    printf("\nAlgorithm A (loop)    : S = %lld   using %lld additions\n", sumA, stepsA);
    printf("Algorithm B (formula) : S = %lld   using %lld calculation\n", sumB, stepsB);
    printf("Post-condition check  : %s\n", sumA == sumB ? "OK, both answers match" : "FAILED");
    return 0;
}
