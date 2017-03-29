#include <math.h>

#include "stuff.h"

void compute(int n, double *input, double *output)
{
    int j;
    for (j = 0; j < n; ++j) {
        output[j] = sin(input[j]);
    }
}
