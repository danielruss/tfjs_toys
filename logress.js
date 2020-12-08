// Propagate is a single step of the forward backward propagate 
// for logistic regression.  Based on the Andrew Ng Coursera course
// implemented in tensorflow.js instead of numpy.

// w is the weights tensor
// b is the bias (scaler)
// x is the traning data (2-d tensor)
// y is the labels (1-d tensor)....
async function propagate(w, b, x, y) {
    [n, m] = x.shape

    // Forward Propagation step 
    let z = tf.matMul(w.transpose(), x).add(b)
    let a = tf.sigmoid(z)
    // For my example data (from Andrew Ng's class), binaryCrossentropy is not  giving me the values I expect
    // actually not sure why but but even my formula also is slightly off....
    // in the example the cost is 5.801545319394553, 
    // binaryCrossentropy gives: 4.8716006
    // and my cost is: 5.798590183258057
    //let cost = tf.metrics.binaryCrossentropy(y, a)
    let cost = tf.mul(-1, tf.mean(tf.add(tf.mul(y, tf.log(a)), tf.mul(tf.sub(1, y), tf.log(tf.sub(1, a))))));
    tf.dispose(z)

    // backwards propagation

    // dw = partials with respect to w
    let dw = tf.matMul(x, tf.sub(a, y).transpose()).div(m)
    // db = partials with respect to b (make a scaler)
    let db = (await tf.sum(tf.sub(a, y)).div(m).data())[0]
    // make the cost a scaler
    cost = (await cost.data())[0]
    tf.dispose(a)
    return ([dw, db, cost])
}


//dw	[[ 0.99845601] [ 2.39507239]]
//db	0.00145557813678
//cost	5.801545319394553

//def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
async function optimize(w, b, x, y, num_iterations, learning_rate, print_cost = false) {
    // run the gradient descent...

    costs = []
    // ugh... loops
    for (let i = 0; i < num_iterations; i++) {
        [dw, db, cost] = await propagate(w, b, x, y)
        // w= w-lr*dw
        // b = b-lr*db <- note b is a scaler
        w = tf.sub(w, tf.mul(learning_rate, dw))
        b = b - learning_rate * db

        if (i % 100 == 0) {
            costs.push(cost)
            if (print_cost) console.log(`${i}: ${cost}`)
        }
    }
    console.log()
    return ([w, b, costs])
}