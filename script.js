const URL = "./my_model/";
let model, webcam, ctx, labelContainer, maxPredictions;

let previousActiveGesture = null;

async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    // load the model and metadata
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Convenience function to setup a webcam
    const widthSize = window.innerWidth;
    const heightSize = window.innerHeight;

    // const widthSize = 1800;
    // const heightSize = 1000;

    const flip = true; // whether to flip the webcam
    webcam = new tmPose.Webcam(widthSize, heightSize, flip); // width, height, flip
    await webcam.setup(); // request access to the webcam
    await webcam.play();
    window.requestAnimationFrame(loop);

    // append/get elements to the DOM
    const canvas = document.getElementById("canvas");
    canvas.width = widthSize;
    canvas.height = heightSize;
    ctx = canvas.getContext("2d");
    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < maxPredictions; i++) { // and class labels
        labelContainer.appendChild(document.createElement("div"));
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
}

async function loop(timestamp) {
    webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Prediction #1: run input through posenet
    // estimatePose can take in an image, video or canvas html element
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Prediction 2: run input through teachable machine classification model
    const prediction = await model.predict(posenetOutput);

    let activeGesture = null;
    let activeGestureValue = 0;

    for (let i = 0; i < maxPredictions; i++) {
        const predictValue = prediction[i].probability.toFixed(2);
        const classPrediction = prediction[i].className + ": " + predictValue;
        labelContainer.childNodes[i].innerHTML = classPrediction;

        if (predictValue > activeGestureValue) {
            activeGesture = prediction[i].className;
            activeGestureValue = predictValue;
        }

        if (activeGesture !== previousActiveGesture && activeGestureValue > 0.8) {
            console.log('activeGesture', activeGesture, activeGestureValue, previousActiveGesture)
            playMusic(activeGesture);
            previousActiveGesture = activeGesture;
        }
    }

    drawPose(pose);
}

function playMusic(pose) {
    const audio = new Audio();

    if (pose === 'pose1') {
        audio.src = './tracks/audio_1.mp3';
    } else if (pose === 'pose2') {
        audio.src = './tracks/audio_2.mp3';
    } else if (pose === 'pose3') {
        audio.src = './tracks/audio_3.mp3';
    } else if (pose === 'pose4') {
        audio.src = './tracks/audio_4.mp3';
    } else if (pose === 'pose5') {
        audio.src = './tracks/audio_5.mp3';
    } else if (pose === 'pose6') {
        audio.src = './tracks/audio_6.mp3';
    }

    console.log('audio', audio)
    // audio.play();
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        // draw the keypoints and skeleton
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}

function resizeCanvas() {
    const canvas = document.getElementById("canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx = canvas.getContext("2d");
}
