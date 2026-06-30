export type TrainingStatus =
  | 'idle'
  | 'connecting'
  | 'running'
  | 'stopped'
  | 'complete'
  | 'error'

export type SchedulerType = 'none' | 'step' | 'cosine' | 'plateau' | 'exponential'

export interface TrainingConfig {
  modelType: 'nn' | 'xgboost'
  dataset: 'mnist' | 'fashion_mnist' | 'cifar10' | 'custom' | 'image_folder'
  epochs: number
  batchSize: number
  learningRate: number
  optimizer: 'adam' | 'sgd' | 'rmsprop'
  lossFunction: 'cross_entropy' | 'mse'
  // Scheduler
  scheduler: SchedulerType
  schedulerStepSize: number
  schedulerGamma: number
  schedulerTMax: number
  // XGBoost hyperparameters
  xgbTask: 'classification' | 'regression'
  xgbObjective: string       // e.g. 'binary:logistic', 'reg:squarederror'
  xgbNEstimators: number
  xgbMaxDepth: number
  xgbLearningRate: number
  xgbSubsample: number
  xgbColsampleBytree: number
  xgbMinChildWeight: number
  xgbGamma: number
  xgbRegAlpha: number
  xgbRegLambda: number
  xgbEarlyStoppingRounds: number
}

export interface EpochMetrics {
  epoch: number
  trainLoss: number
  valLoss: number
  valAccuracy: number
  top5Accuracy?: number
  currentLR?: number
}

export interface TrainingMessage {
  type: string
  [key: string]: unknown
}

export const DEFAULT_CONFIG: TrainingConfig = {
  modelType: 'nn',
  dataset: 'mnist',
  epochs: 10,
  batchSize: 64,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'cross_entropy',
  scheduler: 'none',
  schedulerStepSize: 5,
  schedulerGamma: 0.5,
  schedulerTMax: 10,
  xgbTask: 'classification',
  xgbObjective: '',
  xgbNEstimators: 200,
  xgbMaxDepth: 6,
  xgbLearningRate: 0.1,
  xgbSubsample: 0.8,
  xgbColsampleBytree: 0.8,
  xgbMinChildWeight: 1,
  xgbGamma: 0.0,
  xgbRegAlpha: 0.0,
  xgbRegLambda: 1.0,
  xgbEarlyStoppingRounds: 20,
}

export interface CustomDatasetPayload {
  X_train: number[][]
  y_train: number[]
  X_val: number[][]
  y_val: number[]
  featureCount: number
  classCount: number
  featureNames: string[]
  classNames: string[]
  datasetName: string
  trainSamples: number
  valSamples: number
}

// ── Computer Vision dataset ───────────────────────────────────────────────────

export interface CVDataset {
  sessionId: string
  name: string
  classNames: string[]
  classCounts: Record<string, number>
  totalImages: number
  inputShape: [number, number, number]  // [C, H, W]
  thumbnails: Record<string, string[]>  // class → base64 jpeg thumbnails
  warnings: string[]
}
