#include <stdlib.h>
#include "flappy.h"
#include "puffernet.h"

void demo() {
    Flappy env = {.rng = 42};
    env.observations = (float*)calloc(6, sizeof(float));  // bumped from 1 → 6
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    Weights* weights = load_weights("resources/flappy/flappy_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(weights, 1, 6, 128, 1, logit_sizes, 1);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0.0f;
            if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_W)) env.actions[0] = FLAP;
        } else {
            forward_puffernet(net, env.observations, env.actions);
        }

        c_step(&env);
        c_render(&env);
    }

    free_puffernet(net);
    free(weights);

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main() {
    demo();
    return 0;
}