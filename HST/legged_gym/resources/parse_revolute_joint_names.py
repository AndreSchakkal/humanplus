import xml.etree.ElementTree as ET

def get_revolute_joints(urdf_file):
    """
    Extracts all joint names with the type 'revolute' from a URDF file.

    Args:
        urdf_file (str): Path to the URDF file.

    Returns:
        list: A list of joint names with type 'revolute'.
    """
    tree = ET.parse(urdf_file)  # Parse the URDF file
    root = tree.getroot()       # Get the root element

    revolute_joints = []

    # Iterate through all joint elements
    for joint in root.findall("joint"):
        joint_type = joint.get("type")  # Get the type attribute of the joint
        if joint_type == "revolute":
            joint_name = joint.get("name")  # Get the name attribute of the joint
            revolute_joints.append(joint_name)

    return revolute_joints


def write_revolute_joints_to_file(joint_names, output_file):
    """
    Writes the joint names to a file in the specified format.

    Args:
        joint_names (list): List of revolute joint names.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as file:
        for joint_name in joint_names:
            file.write(f"'{joint_name}' : 0.,\n")


# Usage example
urdf_file = "resources/robots/g1_description/g1_fixed_base_29.urdf"  # Replace with the path to your URDF file
output_file = "revolute_joints.txt"       # Output file path

# Extract revolute joints and write to file
revolute_joints = get_revolute_joints(urdf_file)
write_revolute_joints_to_file(revolute_joints, output_file)

print(f"Revolute joints written to {output_file}")
