async function getData() {
    const heartDataResponse = await
        fetch('https://bernardo0729.github.io/Coracao/coracao.json');
    const heartData = await heartDataResponse.json();
    const cleaned = heartData.map(heart => ({
        SkinThickness: heart.SkinThickness,
        BMI: heart.BMI,
    })).filter(heart => (heart.SkinThickness != null && heart.BMI != null));

    return cleaned;
}

async function run() {
    //carrega e plota os dados de entrada originais nos quais vamos treinar
    const data = await getData();
    const values = data.map(d => ({
        x: d.BMI,
        y: d.SkinThickness
    }));

    tfvis.render.scatterplot(
        { name: 'BMI vs SkinThickness' },
        { values },
        {
            xLabel: 'BMI',
            yLabel: 'SkinThickness',
            height: 300
        }
    );
    //Mais código será adicionando abaixo
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    await trainModel(model, inputs, labels);
    console.log("Treino Completo!");
    // faz algumas previsões usando o modelo e compara com os dados originais
    testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);

function createModel() {
    // Cria um modelo sequencial
    const model = tf.sequential();

    //Adiciona uma unica camada de entrada
    model.add(tf.layers.dense({ inputShape: [1], units: 50, useBias: true }));

    model.add(tf.layers.dense({ units: 50, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 50, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 50, activation: 'relu' }));

    //Adiciona uma camada de saida
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        //Passo 1. Embaralhe os dados
        tf.util.shuffle(data);

        // Etapa 2. Converta dados em tensor
        const inputs = data.map((d) => d.BMI);
        const labels = data.map((d) => d.SkinThickness);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);


        //Etapa 3. Normalize os dados para o intervalo 0 -1 usando  escala min-max
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normallizedInputs = inputTensor
            .sub(inputMin)
            .div(inputMax.sub(inputMin));

        const normallizedLabel = labelTensor
            .sub(labelMin)
            .div(labelMax.sub(labelMin));

        return {
            inputs: normallizedInputs,
            labels: normallizedLabel,
            //Retorne os limites minimo/ máximo para que possamos usá-los
            inputMax,
            inputMin,
            labelMax,
            labelMin,//talvez tire a virgula
        };
    });
}
//exibe o modelo
//tfvis.show.modelSummary((name: "Modelo"), model);

async function trainModel(model, inputs, labels) {
    //prepara o modelo para o treinamento.
    model.compile({
        optimizer: tf.train.adam(),//serve para ajustar os pesos(erros)
        loss: tf.losses.meanSquaredError,// serve de qual certo está errado nossa predição
        metrics: ["mse"],
    });
    const batchSize = 32;
    const epochs = 100;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: "Performance do trainamento" },
            ["loss", "mse"],
            { height: 200, callbacks: ["onEpochEnd"] }
        ),
    });
}

function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));
        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoint = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] };
    });

    const originalPoints = inputData.map((d) => ({
        x: d.BMI,
        y: d.SkinThickness,
    }));

    tfvis.render.scatterplot(
        { name: "Previsões vs Dados originais" },
        {
            values: [originalPoints, predictedPoint],
            series: ["original", "predicted"],
        },
        {
            xLabel: "BMI",
            yLabel: "SkinThickness",
            height: 300,
        }
    );
}

const model = createModel();
tfvis.show.modelSummary({ name: "Modelo" }, model);