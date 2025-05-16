#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <iostream>
#include "chessnet_evaluator.h"

// Abstract evaluator interface
class Evaluator {
public:
    virtual ~Evaluator() = default;
    
    // Evaluate a position and return a score
    // Positive values are good for white, negative for black
    virtual float eval(const std::array<uint64_t, 12>& bitboards) = 0;
    
    // Get a description of the evaluator
    virtual std::string getDescription() const = 0;
};

// Simple material-based evaluator
class MaterialEvaluator : public Evaluator {
private:
    // Material values for pieces (index 0-5: white pieces, 6-11: black pieces)
    // Pawn, Knight, Bishop, Rook, Queen, King
    std::array<int, 12> material_values;

public:
    MaterialEvaluator() {
        // Initialize piece values
        material_values = {
            100, 320, 330, 500, 900, 20000,  // White pieces
            -100, -320, -330, -500, -900, -20000  // Black pieces (negative as they're good for black)
        };
    }
    
    float eval(const std::array<uint64_t, 12>& bitboards) override {
        float score = 0.0f;
        
        // Sum material for all pieces
        for (int i = 0; i < 12; i++) {
            int piece_count = __builtin_popcountll(bitboards[i]);
            score += piece_count * material_values[i];
        }
        
        return score;
    }
    
    std::string getDescription() const override {
        return "Material evaluation";
    }
};

// Re-use the existing neural network evaluator (forward declaration)
class ChessNetEvaluator;

// Neural network evaluator adapter
class NeuralNetEvaluator : public Evaluator {
private:
    ChessNetEvaluator* neural_evaluator;
    bool owns_evaluator;

public:
    NeuralNetEvaluator(const std::string& model_path) 
        : neural_evaluator(new ChessNetEvaluator(model_path)), owns_evaluator(true) {}
    
    // Use an existing ChessNetEvaluator
    NeuralNetEvaluator(ChessNetEvaluator* existing_evaluator) 
        : neural_evaluator(existing_evaluator), owns_evaluator(false) {}
    
    ~NeuralNetEvaluator() {
        if (owns_evaluator && neural_evaluator) {
            delete neural_evaluator;
        }
    }
    
    float eval(const std::array<uint64_t, 12>& bitboards) override {
        return neural_evaluator->eval(bitboards);
    }
    
    std::string getDescription() const override {
        return "Neural network evaluation";
    }
};
