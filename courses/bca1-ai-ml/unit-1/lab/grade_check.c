/*
 * Unit 1 lab · Program 3: specification, validation and test cases
 *
 * Problem : convert marks of one subject into a letter grade
 * Input   : marks, a whole number
 * Output  : the grade  80-100: A   60-79: B   50-59: C   40-49: D   0-39: F
 * Pre     : 0 <= marks <= 100   (checked below, not assumed)
 * Post    : exactly one grade from {A, B, C, D, F} is printed for valid marks;
 *           invalid input prints an error and no grade
 *
 * Try these boundary test cases: -1 0 39 40 49 50 59 60 79 80 100 101 abc
 *
 * Compile : gcc grade_check.c -o grade_check
 */
#include <stdio.h>

int main(void)
{
    int marks;
    char grade;

    printf("Enter marks (0 to 100): ");
    if (scanf("%d", &marks) != 1) {                 /* type check  */
        printf("Error: marks must be a whole number.\n");
        return 1;
    }
    if (marks < 0 || marks > 100) {                 /* range check */
        printf("Error: marks must be between 0 and 100, you entered %d.\n", marks);
        return 1;
    }

    /* pre-condition now holds: 0 <= marks <= 100 */
    if (marks >= 80)      grade = 'A';
    else if (marks >= 60) grade = 'B';
    else if (marks >= 50) grade = 'C';
    else if (marks >= 40) grade = 'D';
    else                  grade = 'F';

    printf("Marks %d -> Grade %c\n", marks, grade);
    return 0;
}
