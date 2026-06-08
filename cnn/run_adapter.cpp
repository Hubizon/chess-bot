#include <unistd.h>
#include <iostream>

int main() {
    const char* bot_prog = "./dist/run_bot/run_bot";
    // Modify the path to the model below
    char* args[] = {(char*)"run_bot", (char*)"--model", (char*)"./model_20260607_223010_22.model", NULL};
    execvp(bot_prog, args);
    std::cerr << "Bot execution failed." << std::endl;
    return 1;
}