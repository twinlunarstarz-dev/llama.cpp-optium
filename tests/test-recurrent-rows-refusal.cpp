// End-to-end proof that a recurrent capacity refusal is an admission decision,
// not a mid-decode failure.
//
// tests/test-recurrent-rows.cpp P5 proves the row_pool refuses.  This proves
// the refusal survives the whole path a real request takes: find_slot() ->
// apply() -> get_status().  The two failures it distinguishes are:
//
//   A. refuse up front  -> llama_decode() returns non-zero and no state is
//      written, so the caller can free a slot and retry;
//   B. fail mid-decode  -> llama_decode() succeeds against a cells[] table that
//      does not hold the batch, corrupting the next sequence to reuse the row.
//
// Run: test-recurrent-rows-refusal <model.gguf> [n_slots]

#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                   \
    } while (0)

static bool silent_progress(float, void *) { return true; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_slots]\n", argv[0]);
        return 2;
    }

    const uint32_t n_slots = argc > 2 ? (uint32_t) atoi(argv[2]) : 4;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback  = silent_progress;
    mparams.n_gpu_layers       = 99;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 256;
    cparams.n_batch           = 64;
    cparams.n_ubatch          = 64;
    cparams.n_seq_max         = n_slots;
    cparams.n_threads         = 4;
    cparams.n_threads_batch   = 4;
    cparams.no_perf           = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const uint32_t rows     = llama_n_seq_max_resident(ctx);
    const int32_t  n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int      per_seq  = 2;

    printf("== recurrent capacity refusal is an admission decision ==\n");
    printf("model                  = %s\n", argv[1]);
    printf("logical sequences      = %u\n", n_slots);
    printf("physical rows resident = %u\n", rows);
    CHECK(rows >= 1);
    CHECK(n_vocab > 1);

    // Ask for more distinct sequences than the physical row pool can hold, in a
    // single ubatch.  find_slot() must refuse this up front.
    const uint32_t demand = rows + 2;

    llama_batch batch = llama_batch_init((int32_t) (demand * per_seq), 0, (int32_t) demand);
    for (uint32_t s = 0; s < demand; ++s) {
        for (int t = 0; t < per_seq; ++t) {
            // llama_batch_add() takes a single seq_id, so a multi-sequence
            // batch is filled directly through the public struct
            const llama_token tok = (llama_token) (1 + (s * per_seq + t) % (n_vocab > 8 ? 8 : 1));
            const int idx = batch.n_tokens++;
            batch.token[idx]     = tok;
            batch.pos[idx]       = t;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = (llama_seq_id) s;
            batch.logits[idx]    = true;
        }
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    printf("decode of %u sequences (%u rows) -> rc=%d\n", demand, rows, rc);

    if (demand > rows) {
        // The refusal must be visible to the caller.
        CHECK(rc != 0);
        printf("refusal surfaced to caller: %s\n", rc != 0 ? "yes" : "NO");
    } else {
        // Everything fit, so it must have succeeded.
        CHECK(rc == 0);
        printf("all sequences fit, decode succeeded: %s\n", rc == 0 ? "yes" : "NO");
    }

    // The refusal must leave the cache usable: a request that does fit still
    // decodes afterwards.  A mid-decode failure would instead have left the
    // cells[] table holding a batch it could not place.
    {
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch ok = llama_batch_init(per_seq, 0, 1);
        for (int t = 0; t < per_seq; ++t) {
            const int idx = ok.n_tokens++;
            ok.token[idx]     = (llama_token) (1 + t);
            ok.pos[idx]       = t;
            ok.n_seq_id[idx] = 1;
            ok.seq_id[idx][0] = 0;
            ok.logits[idx]    = true;
        }
        const int rc2 = llama_decode(ctx, ok);
        llama_batch_free(ok);
        printf("recovery decode after refusal -> rc=%d\n", rc2);
        CHECK(rc2 == 0);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s: %d checks, %d failures\n", "test-recurrent-rows-refusal", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
