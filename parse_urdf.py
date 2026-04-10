import xml.etree.ElementTree as ET

tree = ET.parse("robots/catbot_leg_description/urdf/robot.urdf")
root = tree.getroot()

for joint in root.findall('joint'):
    name = joint.attrib.get('name')
    parent = joint.find('parent').attrib.get('link') if joint.find('parent') is not None else None
    child = joint.find('child').attrib.get('link') if joint.find('child') is not None else None
    origin = joint.find('origin')
    xyz = origin.attrib.get('xyz') if origin is not None else None
    print(f"Joint: {name}")
    print(f"  Parent: {parent}")
    print(f"  Child:  {child}")
    print(f"  xyz:    {xyz}")
