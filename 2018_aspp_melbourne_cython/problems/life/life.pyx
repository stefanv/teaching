cimport cython

@cython.boundscheck(False)
def life_update(int[:, ::1] old_state,
                int[:, ::1] new_state):

    ...
    ...
    # Modify new_state to have a value of 1 (for life) or 0 otherwise

    # No return statement required, modifications are all in-place
