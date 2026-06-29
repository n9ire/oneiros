export type TrainingStatus =
  | 'idle'
  | 'connecting'
  | 'running'
  | 'stopped'
  | 'complete'
  | 'error'

export type SchedulerType = 'none' | 'step' | 'cosine' | 'plateau' | 'exponential'

export interface TrainingConfig {
  dataset: 'mnist' | 'fashion_mnist' | 'cifar10' | 'custom'
  epochs: number
  batchSize: number
  learningRate: number
  optimizer: 'adam' | 'sgd' | 'rmsprop'
  lossFunction: 'cross_entropy' | 'mse'
  // Scheduler
  scheduler: SchedulerType
  schedulerStepSize: number   // StepLR
  schedulerGamma: number      // StepLR / ExponentialLR
  schedulerTMax: number       // CosineAnnealingLR
}

export interface EpochMetrics {
  epoch: number
  trainLoss: number
  valLoss: number
  valAccuracy: number
  currentLR?: number
}

export interface TrainingMessage {
  type: string
  [key: string]: unknown
}

export const DEFAULT_CONFIG: TrainingConfig = {
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
