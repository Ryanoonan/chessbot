#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <chrono>
#include <thread>

class UCI_Engine {
private:
    FILE* engine_process;
    std::string engine_path;
    bool is_initialized;
    int skill_level;

    // Read from engine's stdout
    std::string read_line() {
        if (!engine_process) {
            throw std::runtime_error("Engine process not initialized");
        }

        char buffer[4096];
        if (fgets(buffer, sizeof(buffer), engine_process) == nullptr) {
            return "";
        }
        return std::string(buffer);
    }

    // Read until we get a line containing the specified text
    std::string read_until_contains(const std::string& text, int timeout_ms = 5000) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string result;
        
        while (true) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            
            if (elapsed > timeout_ms) {
                std::cerr << "Timeout waiting for response containing: " << text << std::endl;
                return result;
            }
            
            std::string line = read_line();
            if (!line.empty()) {
                result += line;
                if (line.find(text) != std::string::npos) {
                    return result;
                }
            }
            
            // Small delay to prevent CPU overuse
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:
    UCI_Engine(const std::string& path, int level = 20) 
        : engine_path(path), is_initialized(false), skill_level(level) {
        // Don't start process in constructor
    }
    
    ~UCI_Engine() {
        if (engine_process) {
            // Send quit command
            write_command("quit");
            // Close pipe
            pclose(engine_process);
        }
    }
    
    bool initialize() {
        // Open pipe to stockfish process
        #ifdef _WIN32
        engine_process = _popen(engine_path.c_str(), "r+");
        #else
        // Don't redirect stderr to /dev/null as it can cause segmentation faults
        engine_process = popen(engine_path.c_str(), "r+");
        #endif
        
        if (!engine_process) {
            std::cerr << "Failed to start engine at: " << engine_path << std::endl;
            return false;
        }
        
        // Set to line buffering for faster response
        setbuf(engine_process, nullptr);
        
        // Initialize UCI mode
        write_command("uci");
        read_until_contains("uciok");
        
        // Set skill level (0-20, with 20 being the strongest)
        write_command("setoption name Skill Level value " + std::to_string(skill_level));
        
        // Let engine know we're ready
        write_command("isready");
        read_until_contains("readyok");
        
        is_initialized = true;
        return true;
    }
    
    void write_command(const std::string& command) {
        if (!engine_process) {
            throw std::runtime_error("Engine process not initialized");
        }
        
        // Send command to engine
        fprintf(engine_process, "%s\n", command.c_str());
        fflush(engine_process);
    }
    
    void set_position(const std::string& fen, const std::vector<std::string>& moves = {}) {
        std::stringstream ss;
        ss << "position";
        
        if (fen == "startpos") {
            ss << " startpos";
        } else {
            ss << " fen " << fen;
        }
        
        if (!moves.empty()) {
            ss << " moves";
            for (const auto& move : moves) {
                ss << " " << move;
            }
        }
        
        write_command(ss.str());
    }
    
    std::string get_best_move(int think_time_ms = 10) {
        write_command("go movetime " + std::to_string(think_time_ms));
        
        std::string output = read_until_contains("bestmove");
        
        // Extract the best move from the output
        size_t pos = output.find("bestmove");
        if (pos != std::string::npos) {
            std::string bestmove_line = output.substr(pos);
            std::istringstream iss(bestmove_line);
            std::string token;
            iss >> token; // "bestmove"
            iss >> token; // the actual move
            return token;
        }
        
        return "";
    }
    
    bool is_ready() const {
        return is_initialized;
    }
};
