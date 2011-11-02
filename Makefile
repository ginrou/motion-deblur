#Makefile

MAINFILE = main.c
TARGET   = ${MAINFILE:.c=.out}
SRCS = ${MAINFILE} pcg.c compressiveSensing.c
OBJS := ${SRCS:.c=.o}

CC = gcc
CVFLAGS = `pkg-config --cflags opencv` 
CVLIBS  = `pkg-config --libs opencv`
CFLAGS   = -std=c99

${TARGET}:${OBJS}
	${CC} ${CFLAGS} -o $@ ${OBJS} ${CVFLAGS}  ${CVLIBS} -lm

.c.o:
	${CC} $< ${CFLAGS} -c -o $@  ${CVFLAGS}


clean:
	rm -f ${OBJS}
