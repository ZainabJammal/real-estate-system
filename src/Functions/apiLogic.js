import { useQuery } from "@tanstack/react-query";

const fetchData = async (endpoint, used_method = "GET") => {
  try {
    const res = await fetch(`http://127.0.0.1:8000/${endpoint}`, {
      method: used_method,
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(res.error.message || "Something went wrong");
    }

    return data;
  } catch (error) {
    console.error("Cannot Fetch: ", error);
    throw error;
  }
};

export const useAggValues = () => {
  return useQuery({
    queryKey: ["AggValues"],
    queryFn: () => fetchData("agg_values"),
  });
};

export const useAllLists = () => {
  return useQuery({
    queryKey: ["AllLists"],
    queryFn: () => fetchData("all_lists"),
  });
};

export const useProvList = () => {
  return useQuery({
    queryKey: ["ProvLists"],
    queryFn: () => fetchData("provinces"),
  });
};

export const useHotAreas = () => {
  return useQuery({
    queryKey: ["HotAreas"],
    queryFn: () => fetchData("hot_areas"),
  });
};
