#include <unistd.h>
#include <iostream>

int main() {
    const char* bot_prog = /*INSERT BOT BINARY PATH HERE*/;
    // Modify the path to the model below
    char* args[] = {(char*)"run_bot", (char*)"--model", (char*)/*INSERT MODEL PATH HERE*/, (char*)"--extended", (char*)"True", (char*)"--resnet", (char*)"True", NULL};
    execvp(bot_prog, args);
    std::cerr << "Bot execution failed." << std::endl;
    return 1;
}