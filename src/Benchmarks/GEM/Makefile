include Makefile.inc

SUBDIRS = 
HEADERS = "./TNL/src/TNL"
SOURCES = tml-gem.cpp
CUDA_SOURCES = tnl-gem-cuda.cu
TARGETS = tnl-gem 
CUDA_TARGETS = tnl-gem-cuda

FILES = Makefile \
        Makefile.inc \
        $(CUDA_SOURCES) \
        $(SOURCES) \
        $(HEADERS)

SUBDIRSCLEAN=$(addsuffix clean,$(SUBDIRS))

all: bin subdirs $(TARGETS) $(CUDA_TARGETS)

.PHONY:	subdirs $(SUBDIRS)
subdirs:    $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@	

bin:
	mkdir -p bin

install: all
	mkdir -p $(INSTALL_DIR)/bin
	cp bin/* $(INSTALL_DIR)/bin

.PHONY:	clean
clean:	$(SUBDIRSCLEAN) clean_curdir

clean_curdir:
	rm -f *.o
	
%clean:	%
	$(MAKE) -C $< clean

dist: clean
	tar zcvf $(PROJECT_NAME)-src.tgz $(SUBDIRS) $(FILES)

#$(TARGETS): % : %.o
#	$(CXX) $(LDFLAGS) -L /home/maty/.openmpi/lib -lmpi -o $@ $< $(LDLIBS)

$(CUDA_TARGETS): % : %.cu.o
	$(CUDA_CXX) $(CUDA_LDFLAGS) -I /home/maty/.openmpi/include -L /home/maty/.openmpi/lib -lmpi -o $@ $< $(CUDA_LDLIBS)

#$(SOURCES:%.cpp=%.o): %.o: %.cpp
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(CUDA_SOURCES:%.cu=%.cu.o): %.cu.o : %.cu
	$(CUDA_CXX) $(CUDA_CPPFLAGS) $(CUDA_CXXFLAGS) -I /home/maty/.openmpi/include -c -o $@ $<
