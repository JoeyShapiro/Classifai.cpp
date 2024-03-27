#include <stdio.h>
#include <stdlib.h>
#include "bert.h"

int main() {
    printf("Hello, World!\n");
    
    // load the bert model
    bert_ctx *ctx = bert_load_from_file("final/ggml-model-f32.bin");
    int n = bert_n_embd(ctx);
    printf("n_embd: %d\n", n);

    bert_test(ctx);

    // bert_eval(ctx, 0, "Hello, World!"); 
    // allocate float array
    float *embeddings = (float *)malloc(n * sizeof(float));
    bert_encode(ctx, 0, "func main() \\n{\\n\\tfmt.Println(\"hello world\")\\n}", embeddings);
    printf("embeddings: %f %f %f %f %f %f\n", embeddings[0], embeddings[1], embeddings[2], embeddings[3], embeddings[4], embeddings[5]);
    bert_free(ctx);
    // free float array
    free(embeddings);

    return 0;
}
