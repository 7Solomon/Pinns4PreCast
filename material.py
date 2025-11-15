from dataclasses import dataclass

@dataclass
class BasePhysics:
    g:float = 9.81  # [m/s^2]
    Temp_ref: float = 298.15  # [K] (25°C)
    R: float = 8.31446261815324  # [J/(mol*K)]

@dataclass
class ConcreteData(BasePhysics):
    cp: float = 8.5e2 # [J/(kg*K)]
    rho: float = 2.4e3  # [kg/m^3]
    k: float = 1.4 # [W/(m*K)]

    Q_pot: float = 500e3  # [J/kg]
    B1: float = 0.0002916  # [1/s]
    B2: float = 0.0024229  # [1/s]
    deg_hydr_max: float = 0.875  # [-]
    eta: float = 5.554  # [-]

    @property
    def Ea(self) -> float:
        return 5.653 * self.R  # [J/mol]

import numpy as np
###CUBE
def cube():
    print("---cube properties---")
    #initial and ambient temperature
    Temp_amb = np.arange(10, 51, 10)+273.15 #ambient temperature [K]
    Temp_init = np.arange(10, 41, 10)+273.15 #initial temperature [K]
    h=10 #Heat transfer coefficient [W /(m² *K)]

    #Material properties
    rho=430+301.5+339.7+335+729.7+193.4+6.8 #density [kg/m³]

    cem=430 #amount of cement in concrete [kg/m³]
    c=2.4E6/rho #specific thermal capacity [J/(kg*K)]
    k=2.6 # Thermal conductivity (W/(m*K))
    R= 8.3145 #ideal gas constant [J/(K*mol)]
    E_a_R =5653 #activation energy [K]
    E_a=E_a_R*R #activation energy [J/mol]
    Temp_ref = 25.+273 #reference temperature [K]

    #Affinity law
    B1=  0.0002916 #[1/s]
    B2=  0.0024229
    deg_hydr_max= .875
    eta=5.554
    Q_pot=500E3 #latent heat of hydration [J/kg]

    print(f"Density: {rho} kg/m³")
    print(f"Cement content: {cem} kg/m³")
    print(f"Q_pot: {Q_pot} J/kg")


##TUNNEL_SEGMENT
def tunnel():
    print("---tunnel segment properties---")
    #constants
    h=10 #Heat transfer coefficient [W /(m² *K)]
    h_side=5.208 #Heat transfer coefficient [W /(m² *K)]
    rho=2507.46197 #density [kg/m³]
    cem=270.7305748 # amount of cement in concrete [kg/m³]
    water=66.1111 # amount of water in concrete [kg/m³]
    wc=water/cem #water to cement ratio
    FA=122.164 # amount of flyash in concrete [kg/m³]
    CaO_FA=0.24 #CaO content in flyash [g/g] assumption based on https://www.fhwa.dot.gov/pavement/recycling/fach01.cfm#:~:text=Chemistry.,based%20on%20its%20chemical%20composition.
    binder=cem+FA #amount of binders in concrete [kg/m³]
    c=2.4E6/rho #specific thermal capacity [J/(kg*K)]
    k=2.6 # Thermal conductivity (W/(m*K))
    R= 8.3145 #ideal gas constant [J/(K*mol)]
    #E_a=38300*(cem/binder)(1-1.05(FA/binder)*(1-(CaO_FA*FA/binder)/0.4)) #activation energy [J/mol]
    Temp_ref = 25.+273 #reference temperature [K]


    #initial and ambient temperature
    Temp_amb = 19.0+273.15 #ambient temperature [K]
    Temp_init = 22.+273.15 #initial temperature [K]

    #Parameters of afﬁnity hydration model for different cement types for Tref =25°C
    # https://doi.org/10.1007/978-3-319-76617-1
    B1= 0.000379 #[1/s]
    B2= 6e-5
    deg_hydr_max= 1.031*wc/(0.194+wc) +0.5*FA/binder #maximum degree of hydration [-] based on https://doi.org/10.14359/14246
    eta=5.8
    Q_pot=(510+1800*FA*CaO_FA/binder)*1E3 #latent heat of hydration [J/kg]

    print(f"Density: {rho} kg/m³")
    print(f"Cement content: {cem} kg/m³")
    print(f"Q_pot: {Q_pot} J/kg")

if __name__ == "__main__":
    cube()
    tunnel()