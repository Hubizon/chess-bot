CXX := g++
CXXFLAGS := -std=c++20

SRCS := $(wildcard *.cpp)
CPPFILES := $(SRCS:.cpp=)

.PHONY: all
all: $(CPPFILES)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(CPPFILES)