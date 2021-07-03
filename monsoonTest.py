import Monsoon.HVPM as MVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op

Mon = MVPM.Monsoon()
Mon.setup_usb()

Mon.setVout(4.0)
engine = sampleEngine.SampleEngine(Mon)
engine.enableCSVOutput("Main Example.csv")
engine.ConsoleOutput(True)
numSamples=5000 #sample for one second
engine.startSampling(numSamples)


engine.disableChannel(sampleEngine.channels.MainCurrent)
engine.disableChannel(sampleEngine.channels.MainVoltage)

engine.enableChannel(sampleEngine.channels.USBCurrent)
engine.enableChannel(sampleEngine.channels.USBVoltage)

Mon.setUSBPassthroughMode(op.USB_Passthrough.On)