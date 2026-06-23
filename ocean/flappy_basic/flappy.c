/* Pure C demo file for Flappy. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "flappy.h"
#include "puffernet.h"

void demo() {
    Flappy env = {.rng = 42};
    env.observations = (float*)calloc(1, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    Weights* weights = load_weights("resources/flappy/flappy_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(weights, 1, 1, 128, 1, logit_sizes, 1);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0.0f;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
        } else {
            float obs_f[1];
            for(int i=0; i<1; i++) obs_f[i] = (float)env.observations[i];
            forward_puffernet(net, obs_f, env.actions);
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