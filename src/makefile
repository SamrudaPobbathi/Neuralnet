CXX = g++
CC = gcc
CFLAGS = -Wall -c
CXXFLAGS = -Wall -D__STDC_LIMIT_MACROS

OUTPUTDIR = ./results/
MKDIR = mkdir -p $(OUTPUTDIR)

CSOURCES = \
     $data_loader.c \

CXXSOURCES = \
			train.cpp

data_loader:
	gcc -c -o data_loader.o data_loader.c \

train:
	g++ -O3 -Wall -c -o train.o train.cpp \
	g++ -o train data_loader.o train.o \
