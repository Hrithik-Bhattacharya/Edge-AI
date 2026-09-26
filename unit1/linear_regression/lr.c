#include <stdio.h>
#include <time.h>

int main() {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {32.0, 41.0, 55.0, 60.0, 74.0};

    int n = 5;
    double w = 0.0;
    double b = 0.0;
    double lr = 0.01;

    clock_t start, end;

    // Start timer
    start = clock();

    for (int step = 0; step <= 3000; step++) {
        double dw = 0.0;
        double db = 0.0;
        double loss = 0.0;

        // Calculate predictions, error, and loss
        for (int i = 0; i < n; i++) {
            double y_hat = w * x[i] + b;
            double err = y_hat - y[i];

            loss += err * err;
            dw += err * x[i];
            db += err;
        }

        // Equivalent to np.mean()
        loss /= n;
        dw = 2.0 * dw / n;
        db = 2.0 * db / n;

        // Gradient descent update
        w -= lr * dw;
        b -= lr * db;

        if (step % 500 == 0) {
            printf("step %4d loss %8.2f w %6.2f b %6.2f\n",
                   step, loss, w, b);
        }
    }

    // Stop timer
    end = clock();

    // Calculate runtime in seconds
    double runtime = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nFinal values:\n");
    printf("w = %.6f\n", w);
    printf("b = %.6f\n", b);
    printf("Runtime = %.9f seconds\n", runtime);

    return 0;
}
