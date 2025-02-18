# Compilers
CC   = mpicc
FC   = mpif90
NVCC = nvcc

# Flags
LFLAGS   += -fPIC -fopenmp
OPTS     += -O2 -DLONGSHORT -DMYALLTOALLV -DPMTHREAD -DPERIODIC_CONDITION

# Directories
SRCDIR    = src
INCDIR    = inc
OBJDIR    = obj
EXEDIR    = ./run/
EXE       = photoNs-lcdm

# Libraries
LIBS     += -fopenmp -lm -lpthread -L/usr/local/cuda/lib64 -lcudart
LIB2DFFT += -L$(DECOMP_INSTALL_DIR)/lib64 -L$(DECOMP_INSTALL_DIR)/lib -ldecomp2d -L"/home/msp/Documents/2decomp-fft/lib/"

# Sources
SOURCES      = photoNs.c domains.c initial.c remotes.c toptree.c utility.c operator.c snapshot.c partmesh.c fmm.c
CONV2D       = conv.f90
CUDA_SOURCES = photoNs_CUDA.cu

# Objects
OBJECTS      = $(patsubst %.c, $(OBJDIR)/%.o, $(SOURCES))
OBJFORT      = $(patsubst %.f90, $(OBJDIR)/%.o, $(CONV2D))
CUDA_OBJECTS = $(patsubst %.cu, $(OBJDIR)/%.o, $(CUDA_SOURCES))

# Include paths
INCLUDES += -I$(INCDIR) -I/usr/local/cuda/include/
INC2DFFT += -I$(INCDIR)

# Rules
$(OBJDIR)/%.o: $(SRCDIR)/%.f90
	@mkdir -p $(OBJDIR)
	$(FC) -c $(INC2DFFT) -o "$@" "$<"

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) -c $(CFLAGS) $(OPTS) $(INCLUDES) -o "$@" "$<"

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) -c -O2 -I$(INCDIR) $(INCLUDES) -o "$@" "$<"

# Main target
exe: $(OBJECTS) $(OBJFORT) $(CUDA_OBJECTS)
	@mkdir -p $(EXEDIR)
	$(FC) $(OBJECTS) $(OBJFORT) $(CUDA_OBJECTS) $(LIBS) $(LIB2DFFT) -o $(EXEDIR)/$(EXE)

# Demo target
.PHONY: demo
demo:
	cd $(EXEDIR); \
	mpirun -np 1 ./$(EXE) ../demo/lcdm_g2.run #1  #1 specifies printing detailed logs of what is hapenning in GPU exeution

# Clean target
.PHONY: clean
clean:
	rm -rf $(OBJDIR)/*.o $(EXEDIR)/$(EXE)

