#include <unistd.h>
#include <iostream>

int main() {
    const char* bot_prog = "<PATH>/chess-bot/chess-bot/cnn/dist/run_bot/run_bot";
    // Modify the path to the model below
    char* args[] = {(char*)"run_bot", (char*)"--model", (char*)"<PATH>/MPUM/chess-bot/chess-bot/cnn/adapters/model_20260611_022317_extended_8.model", (char*)"--extended", (char*)"True", (char*)"--resnet", (char*)"True", NULL};
    execvp(bot_prog, args);
    std::cerr << "Bot execution failed." << std::endl;
    return 1;
}