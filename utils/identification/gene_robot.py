import sys, pickle
sys.path.append('../..')
from utils.identification.SymPyBotics import sympybotics

def get_robot(path):
    with open(path, 'rb') as file:
        rbt = pickle.load(file)
    return rbt

def create_hand_gx11pm():
    # (alpha, a, d, theta)
    rbtdef = sympybotics.RobotDef("Estun 6 DOF",
                [("pi/2", 0.0305,   -0.0305,   "q"),
                 ("-pi/2", 0.0345,   0.0,   "q"),
                 ("0",    0.0555, 0.0, "q")],
                dh_convention="mdh")

    rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    rbtdef.driveinertiamodel = 'simplified'

    rbt = sympybotics.RobotDynCode(rbtdef)
    rbt.calc_base_parms()

    with open('../../data/model/gx11pm_finger1.pkl', 'wb') as file:
        pickle.dump(rbt, file)

    rbtdef = sympybotics.RobotDef("Estun 6 DOF",
                [("0", 0.096,   0.029,   "q"),
                 ("-pi/2", 0.0515,  -0.0,   "q"),
                 ("pi/2",    0.0, 0.0, "q"),
                 ("pi",    0.036, 0.0, "q")],
                dh_convention="mdh")

    rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    rbtdef.driveinertiamodel = 'simplified'

    rbt = sympybotics.RobotDynCode(rbtdef)
    rbt.calc_base_parms()

    with open('../../data/model/gx11pm_finger2.pkl', 'wb') as file:
        pickle.dump(rbt, file)


    rbtdef = sympybotics.RobotDef("Estun 6 DOF",
                [("0", 0.096,   -0.029,   "q"),
                 ("-pi/2", 0.0515,  -0.0,   "q"),
                 ("pi/2",    0.0, 0.0, "q"),
                 ("pi",    0.036, 0.0, "q")],
                dh_convention="mdh")

    rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    rbtdef.driveinertiamodel = 'simplified'

    rbt = sympybotics.RobotDynCode(rbtdef)
    rbt.calc_base_parms()

    with open('../../data/model/gx11pm_finger3.pkl', 'wb') as file:
        pickle.dump(rbt, file)




def create_hand_gx11():
    # (alpha, a, d, theta)
    rbtdef = sympybotics.RobotDef("Estun 6 DOF",
                [("pi/2", 0.03,   -0.06,   "q"),
                 ("-pi/2", 0.029,   0.0,   "q"),
                 ("0",    0.0505, 0.0, "q")],
                dh_convention="mdh")

    rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    rbtdef.driveinertiamodel = 'simplified'

    rbt = sympybotics.RobotDynCode(rbtdef)
    rbt.calc_base_parms()

    with open('../../data/model/gx11_finger1.pkl', 'wb') as file:
        pickle.dump(rbt, file)
    # rbtdef = sympybotics.RobotDef("Estun 6 DOF",
    #             [("0", 0.096,   0.029,   "q"),
    #              ("-pi/2", 0.0515,  -0.0,   "q"),
    #              ("pi/2",    0.0, 0.0, "q"),
    #              ("pi",    0.036, 0.0, "q")],
    #             dh_convention="mdh")
    rbtdef = sympybotics.RobotDef("Estun 6 DOF",
                [("0", 0.134,   0.029,   "q"),
                 ("pi/2", 0.0515,  -0.0,   "q"),
                 ("-pi/2",    0.0, 0.0, "q"),
                 ("0",    0.036, 0.0, "q")],
                dh_convention="mdh")

    rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    rbtdef.driveinertiamodel = 'simplified'

    rbt = sympybotics.RobotDynCode(rbtdef)
    rbt.calc_base_parms()

    with open('../../data/model/gx11_finger2.pkl', 'wb') as file:
        pickle.dump(rbt, file)


    rbtdef = sympybotics.RobotDef("Estun 6 DOF",
                [("0", 0.134,   -0.029,   "q"),
                 ("pi/2", 0.0515,  -0.0,   "q"),
                 ("-pi/2",    0.0, 0.0, "q"),
                 ("0",    0.036, 0.0, "q")],
                dh_convention="mdh")

    rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    rbtdef.driveinertiamodel = 'simplified'

    rbt = sympybotics.RobotDynCode(rbtdef)
    rbt.calc_base_parms()

    with open('../../data/model/gx11_finger3.pkl', 'wb') as file:
        pickle.dump(rbt, file)





if __name__ == '__main__':
    
    # create_hand_gx11pm()
    create_hand_gx11()

    # rbtdef = sympybotics.RobotDef("Estun 6 DOF",
    #             [("0",       0,   0.44,   "q"),
    #              ("-pi/2",   0,   0.20,   "q-pi/2"),
    #              ("0",     0.460, -0.16, "q"),
    #              ("-pi/2", 0.0,   0.44, "q"),
    #              ("pi/2",  0.0,   0.0,  "q"),
    #              ("-pi/2", 0.0,   0.0,  "q")],
    #             dh_convention="mdh")

    # rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
    # # rbtdef.frictionmodel = {}
    # # rbtdef.frictionmodel = {'Coulomb'}
    # rbtdef.driveinertiamodel = 'simplified'

    # rbt = sympybotics.RobotDynCode(rbtdef)
    # rbt.calc_base_parms()

    # print(dir(rbt))
    # print(len(rbt.dyn.baseparms.n()))

    # with open('../../data/model/estun_model_mdh.pkl', 'wb') as file:
    # 	pickle.dump(rbt, file)











