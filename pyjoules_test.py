from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter

domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
devices = DeviceFactory.create_devices(domains)
meter = EnergyMeter(devices)

def foo():
    for i in range(100):
        pass

def bar():
    for i in range(1000):
        pass

meter.start(tag='foo')
foo()
meter.record(tag='bar')
bar()
meter.stop()

trace = meter.get_trace()