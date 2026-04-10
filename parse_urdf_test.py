import xml.etree.ElementTree as ET
import numpy as np

def extract_4bar_lengths(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    # find joints 'a', 'l' and linkages
    for joint in root.findall('joint'):
        print(joint.attrib['name'])
        
extract_4bar_lengths("robots/catbot_leg_description/urdf/robot.urdf")
