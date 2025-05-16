#pragma once

#include <torch/script.h>
#include <array>
#include <cstdint>
#include <string>
#include <iostream>
#include <vector>

class ChessNetEvaluator {
private:
    torch::jit::script::Module module;
    
public:
    ChessNetEvaluator(const std::string& model_path) {
        try {
            // Load the TorchScript model
            module = torch::jit::load(model_path);
            module.eval();
            std::cout << "Successfully loaded the model from " << model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Converts bitboards to the tensor format expected by the neural network
    torch::Tensor bitboards_to_tensor(const std::array<uint64_t, 12>& bitboards) {
        // Create a tensor with the shape expected by the ChessNet model [64, 64, 10, 2]
        torch::Tensor tensor = torch::zeros({64, 64, 10, 2}, torch::kFloat32);
        
        // Maps piece types to planes (from board_to_tensor_nnue)
        // 0-4: White pawn, knight, bishop, rook, queen
        // 5-9: Black pawn, knight, bishop, rook, queen
        // Kings are special (see below)
        
        // Find king squares
        int white_king_sq = -1;
        int black_king_sq = -1;
        
        for (int sq = 0; sq < 64; sq++) {
            if (bitboards[5] & (1ULL << sq)) { // White king
                white_king_sq = sq;
            }
            if (bitboards[11] & (1ULL << sq)) { // Black king
                black_king_sq = sq;
            }
        }
        
        // Ensure kings are found
        if (white_king_sq == -1 || black_king_sq == -1) {
            std::cerr << "Warning: King not found on the board!" << std::endl;
            return tensor;
        }
        
        // Process pieces for each perspective (white king and black king)
        // First for white king perspective (perspective 0)
        for (int sq = 0; sq < 64; sq++) {
            // Process white pieces (excluding king)
            for (int piece = 0; piece < 5; piece++) { // Pawn, knight, bishop, rook, queen
                if (bitboards[piece] & (1ULL << sq)) {
                    tensor[white_king_sq][sq][piece][0] = 1.0;
                }
            }
            
            // Process black pieces (excluding king)
            for (int piece = 6; piece < 11; piece++) { // Pawn, knight, bishop, rook, queen
                if (bitboards[piece] & (1ULL << sq)) {
                    tensor[white_king_sq][sq][piece-6+5][0] = 1.0; // Offset by 5 for black pieces
                }
            }
        }
        
        // Then for black king perspective (perspective 1)
        for (int sq = 0; sq < 64; sq++) {
            // Process white pieces (excluding king)
            for (int piece = 0; piece < 5; piece++) { // Pawn, knight, bishop, rook, queen
                if (bitboards[piece] & (1ULL << sq)) {
                    tensor[black_king_sq][sq][piece][1] = 1.0;
                }
            }
            
            // Process black pieces (excluding king)
            for (int piece = 6; piece < 11; piece++) { // Pawn, knight, bishop, rook, queen
                if (bitboards[piece] & (1ULL << sq)) {
                    tensor[black_king_sq][sq][piece-6+5][1] = 1.0; // Offset by 5 for black pieces
                }
            }
        }
        
        return tensor;
    }
    
    float eval(const std::array<uint64_t, 12>& bitboards) {
        // Convert bitboards to tensor
        torch::Tensor input_tensor = bitboards_to_tensor(bitboards);
        
        // Create a vector of inputs for the forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        try {
            // Forward pass
            torch::Tensor output = module.forward(inputs).toTensor();
            // Return the scalar evaluation value
            float evaluation = output.item<float>();
            return evaluation;
        } catch (const c10::Error& e) {
            std::cerr << "Error in forward pass: " << e.what() << std::endl;
            return 0.0f;
        }
    }
};

// Legacy class for backward compatibility if needed
class NNUEEvaluator : public ChessNetEvaluator {
public:
    NNUEEvaluator(const std::string& model_path) : ChessNetEvaluator(model_path) {}
};