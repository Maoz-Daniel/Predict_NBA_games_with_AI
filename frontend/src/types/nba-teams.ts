export enum NBATeamLogo {
  HAWKS = "https://upload.wikimedia.org/wikipedia/en/2/24/Atlanta_Hawks_logo.svg",
  CELTICS = "https://upload.wikimedia.org/wikipedia/en/8/8f/Boston_Celtics.svg",
  NETS = "https://upload.wikimedia.org/wikipedia/commons/4/44/Brooklyn_Nets_newlogo.svg",
  HORNETS = "https://upload.wikimedia.org/wikipedia/en/c/c4/Charlotte_Hornets_%282014%29.svg",
  BULLS = "https://upload.wikimedia.org/wikipedia/en/6/67/Chicago_Bulls_logo.svg",
  CAVALIERS = "https://upload.wikimedia.org/wikipedia/commons/0/04/Cleveland_Cavaliers_secondary_logo.svg",
  MAVERICKS = "https://upload.wikimedia.org/wikipedia/en/9/97/Dallas_Mavericks_logo.svg",
  NUGGETS = "https://upload.wikimedia.org/wikipedia/en/7/76/Denver_Nuggets.svg",
  PISTONS = "https://upload.wikimedia.org/wikipedia/commons/4/4e/Logo_of_the_Detroit_Pistons_%282005%E2%80%932017%29.png",
  WARRIORS = "https://upload.wikimedia.org/wikipedia/en/0/01/Golden_State_Warriors_logo.svg",
  ROCKETS = "https://upload.wikimedia.org/wikipedia/en/2/28/Houston_Rockets.svg",
  PACERS = "https://upload.wikimedia.org/wikipedia/en/1/1b/Indiana_Pacers.svg",
  CLIPPERS = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Los_angeles_clippers_logo_1984-2010.jpg",
  LAKERS = "https://upload.wikimedia.org/wikipedia/commons/3/3c/Los_Angeles_Lakers_logo.svg",
  GRIZZLIES = "https://upload.wikimedia.org/wikipedia/en/f/f1/Memphis_Grizzlies.svg",
  HEAT = "https://upload.wikimedia.org/wikipedia/en/f/fb/Miami_Heat_logo.svg",
  BUCKS = "https://upload.wikimedia.org/wikipedia/en/4/4a/Milwaukee_Bucks_logo.svg",
  TIMBERWOLVES = "https://upload.wikimedia.org/wikipedia/en/c/c2/Minnesota_Timberwolves_logo.svg",
  PELICANS = "https://upload.wikimedia.org/wikipedia/en/0/0d/New_Orleans_Pelicans_logo.svg",
  KNICKS = "https://upload.wikimedia.org/wikipedia/en/2/25/New_York_Knicks_logo.svg",
  THUNDER = "https://upload.wikimedia.org/wikipedia/en/5/5d/Oklahoma_City_Thunder.svg",
  MAGIC = "https://upload.wikimedia.org/wikipedia/en/1/10/Orlando_Magic_logo.svg",
  SIXERS = "https://upload.wikimedia.org/wikipedia/en/0/0e/Philadelphia_76ers_logo.svg",
  SUNS = "https://upload.wikimedia.org/wikipedia/en/d/dc/Phoenix_Suns_logo.svg",
  TRAIL_BLAZERS = "https://upload.wikimedia.org/wikipedia/en/2/21/Portland_Trail_Blazers_logo.svg",
  KINGS = "https://upload.wikimedia.org/wikipedia/en/c/c7/SacramentoKings.svg",
  SPURS = "https://upload.wikimedia.org/wikipedia/en/a/a2/San_Antonio_Spurs.svg",
  RAPTORS = "https://upload.wikimedia.org/wikipedia/en/3/36/Toronto_Raptors_logo.svg",
  JAZZ = "https://upload.wikimedia.org/wikipedia/commons/6/65/Utah_Jazz_wordmark_logo_primary_color.png",
  WIZARDS = "https://upload.wikimedia.org/wikipedia/en/0/02/Washington_Wizards_logo.svg"
}


export type NBATeamName = keyof typeof NBATeamLogo;

export function getTeamLogo(teamName: NBATeamName): string {
  return NBATeamLogo[teamName];
}

