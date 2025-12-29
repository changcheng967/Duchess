#include <iostream>
#include <fstream>
#include <iomanip>

int main() {
    std::ifstream file("C:\\Users\\chang\\Downloads\\Duchess\\nn-2962dca31855.nnue", std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        return 1;
    }
    
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    std::cout << "Magic number: 0x" << std::hex << magic << std::dec << std::endl;
    
    // Try to read more header data
    uint32_t version, input_type;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&input_type), sizeof(input_type));
    
    std::cout << "Version: 0x" << std::hex << version << std::dec << std::endl;
    std::cout << "Input type: 0x" << std::hex << input_type << std::dec << std::endl;
    
    file.close();
    return 0;
}