#include <stdio.h>
#include <inttypes.h>

int py_c_access_test(void * void_ptr, int array_size)
{
    int *array_ptr = (int*)void_ptr;
    int i = 0;
    for (i = 0; i < array_size; i++)
    {
        // Python should be passing us a nice range'd array
        if (array_ptr[i] != i)
            return -1;
    }
    return 0;
}

