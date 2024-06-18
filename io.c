#include "io.h"
#include <stdio.h>
#include <assert.h>

int save_nodes_to_file(const char* fname, unsigned n, const double* x, const double* y)
{
    FILE* fout = fopen(fname, "w");
    if (!fout)
    {
        return -1;
    }

    fprintf(fout, "# x y\n");

    for (unsigned i = 0; i < n; ++i)
    {
        fprintf(fout, "%.15e %.15e\n", x[i], y[i]);
    }

    fclose(fout);
    return 0;
}