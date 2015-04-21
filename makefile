CC=gcc
CFLAGS=-shared

LDIR =lib

_TEST_SRC = $(wildcard tests/*.c)
TEST_LIBS = $(patsubst tests/%.c, %.so, $(_TEST_SRC))

_ALG_SRC = $(wildcard algorithms/*.c)
ALG_LIBS = $(patsubst algorithms/%.c, %.so, $(_ALG_SRC))

_UTILS_SRC = $(wildcard utils/*.c)
UTILS_LIBS = $(patsubst utils/%.c, %.so, $(_UTILS_SRC))

all: tests algorithms utils

$(TEST_LIBS):
	$(CC) $(CFLAGS) -o $(LDIR)/$@ $(patsubst %.so, tests/%.c, $@)

$(ALG_LIBS):
	$(CC) $(CFLAGS) -o $(LDIR)/$@ $(patsubst %.so, algorithms/%.c, $@)

$(UTILS_LIBS):
	$(CC) $(CFLAGS) -o $(LDIR)/$@ $(patsubst %.so, utils/%.c, $@)

tests: $(TEST_LIBS)

algorithms: $(ALG_LIBS)

utils: $(UTILS_LIBS)

.PHONY: clean

# delete compiled libraries
clean:
	rm -f $(LDIR)/*.so *~ 
