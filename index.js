const brain = require("brain.js");

const net = new brain.recurrent.LSTMTimeStep({
  inputSize: 2,
  hiddenLayers: [10],
  outputSize: 2,
});

// Same test as previous, but combined on a single set
const trainingData = [
  [
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10],
  ],
];

net.train(trainingData, { log: true, errorThresh: 0.09 });

const forecast = net.forecast(
  [
    [1, 2],
    [2, 4],
    [3, 6],
    // [4, 8],
    // [5, 10],
  ],
  3
);

console.log("next 3 predictions", forecast);
